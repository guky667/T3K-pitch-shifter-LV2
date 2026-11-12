# PitchShift DSP by TONE3000 (https://github.com/tone-3000/tone3000-plugin)
# MOD LV2 recipe built by guky667
# SPDX-License-Identifier: MIT

T3K_PITCH_SHIFT_VERSION = 61d38eb638449647fb8395a35c5b8dab7e981ba7
T3K_PITCH_SHIFT_SITE = https://github.com/DISTRHO/DPF.git
T3K_PITCH_SHIFT_SITE_METHOD = git
T3K_PITCH_SHIFT_BUNDLES = t3k-pitch-shift.lv2

define T3K_PITCH_SHIFT_PLUGIN_CPP
#include "DistrhoPlugin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

START_NAMESPACE_DISTRHO

namespace {

constexpr double kPi = 3.14159265358979323846;

template <typename T>
inline T jlimit(T lower, T upper, T value)
{
    return value < lower ? lower : (value > upper ? upper : value);
}

inline int roundToInt(double x)
{
    return static_cast<int>(std::lround(x));
}

inline bool exactlyEqual(float a, float b)
{
    return a == b;
}

// Standalone replacement for juce::LinearSmoothedValue<float>
struct LinearSmoothedValue
{
    float currentValue = 0.0f;
    float target = 0.0f;
    float step = 0.0f;
    int countdown = 0;
    int stepsToTarget = 0;

    void reset(double sampleRate, double rampLengthInSeconds)
    {
        stepsToTarget = static_cast<int>(std::floor(sampleRate * rampLengthInSeconds));
        setCurrentAndTargetValue(target);
    }

    void setCurrentAndTargetValue(float newValue)
    {
        target = currentValue = newValue;
        countdown = 0;
    }

    void setTargetValue(float newTarget)
    {
        if (newTarget == target)
            return;
        target = newTarget;
        if (stepsToTarget <= 0) {
            currentValue = target;
            countdown = 0;
        } else {
            countdown = stepsToTarget;
            step = (target - currentValue) / static_cast<float>(countdown);
        }
    }

    float getNextValue()
    {
        if (countdown <= 0)
            return target;
        --countdown;
        if (countdown <= 0)
            currentValue = target;
        else
            currentValue += step;
        return currentValue;
    }

    bool isSmoothing() const
    {
        return countdown > 0;
    }
};

enum class LinkwitzRileyFilterType { lowpass, highpass, allpass };

// Standalone TPT SVF replacement for juce::dsp::LinkwitzRileyFilter<float> (LR4)
struct LinkwitzRileyFilter
{
    static constexpr int kMaxCh = 2;
    LinkwitzRileyFilterType type = LinkwitzRileyFilterType::lowpass;
    double sampleRate = 48000.0;
    float cutoffFrequency = 2000.0f;
    float g = 0.0f;
    float R2 = 1.4142135623730951f;
    float h = 0.0f;
    std::array<float, kMaxCh> s1{}, s2{}, s3{}, s4{};

    void prepare(double sr)
    {
        sampleRate = sr > 0.0 ? sr : 48000.0;
        s1.fill(0.0f);
        s2.fill(0.0f);
        s3.fill(0.0f);
        s4.fill(0.0f);
        updateCoefficients();
    }

    void setType(LinkwitzRileyFilterType newType)
    {
        type = newType;
    }

    void setCutoffFrequency(float newCutoffFrequencyHz)
    {
        cutoffFrequency = newCutoffFrequencyHz;
        updateCoefficients();
    }

    void updateCoefficients()
    {
        g = static_cast<float>(std::tan(kPi * static_cast<double>(cutoffFrequency) / sampleRate));
        R2 = 1.4142135623730951f;
        h = 1.0f / (1.0f + R2 * g + g * g);
    }

    float processSample(int ch, float x)
    {
        const float yH1 = (x - (R2 + g) * s1[ch] - s2[ch]) * h;
        const float tB1 = g * yH1;
        const float yB1 = tB1 + s1[ch];
        s1[ch] = tB1 + yB1;
        const float tL1 = g * yB1;
        const float yL1 = tL1 + s2[ch];
        s2[ch] = tL1 + yL1;

        if (type == LinkwitzRileyFilterType::allpass)
            return yL1 - R2 * yB1 + yH1;

        const float in2 = (type == LinkwitzRileyFilterType::lowpass) ? yL1 : yH1;
        const float yH2 = (in2 - (R2 + g) * s3[ch] - s4[ch]) * h;
        const float tB2 = g * yH2;
        const float yB2 = tB2 + s3[ch];
        s3[ch] = tB2 + yB2;
        const float tL2 = g * yB2;
        const float yL2 = tL2 + s4[ch];
        s4[ch] = tL2 + yL2;

        return (type == LinkwitzRileyFilterType::lowpass) ? yL2 : yH2;
    }
};

// Tuning constants from TONE3000 PitchShift.cpp
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

} // namespace

// TONE3000 PitchShift engine
class PitchShift
{
public:
    static constexpr int kMaxChannels = 2;
    static constexpr int kSemitoneRange = 24;
    static constexpr float kTonalityMinHz = 1000.0f;
    static constexpr float kTonalityOffHz = 20000.0f;

    enum class Window { ms20 = 0, ms30 = 1, ms40 = 2, ms60 = 3 };
    static constexpr Window kDefaultWindow = Window::ms30;
    static constexpr double kMinDelayMs = 2.0;

    static int windowMs(Window w)
    {
        switch (w) {
        case Window::ms20: return 20;
        case Window::ms30: return 30;
        case Window::ms40: return 40;
        case Window::ms60: return 60;
        }
        return 30;
    }

    static Window windowFromIndex(int index)
    {
        return static_cast<Window>(jlimit(0, 3, index));
    }

    struct Params
    {
        float semitones = 0.0f;
        float tonalityHz = 0.0f;
        Window window = kDefaultWindow;
        float mix = 1.0f;
    };

    static int minDelaySamples(double sampleRate)
    {
        return static_cast<int>(sampleRate * kMinDelayMs * 0.001);
    }

    static int windowSamples(Window w, double sampleRate)
    {
        return static_cast<int>(sampleRate * windowMs(w) * 0.001);
    }

    void prepare(double sampleRate)
    {
        auto& s = impl_;
        s.sampleRate = sampleRate;
        s.fadeLen = std::max(8, static_cast<int>(sampleRate * kFadeMs * 0.001));
        s.fadeMaxLen = std::max(s.fadeLen, static_cast<int>(sampleRate * kFadeMaxMs * 0.001));
        s.fadeMinLen = std::max(8, static_cast<int>(sampleRate * kFadeMinMs * 0.001));
        s.onsetFadeLen = std::max(8, static_cast<int>(sampleRate * kOnsetFadeMs * 0.001));
        s.onsetSpan = static_cast<int>(sampleRate * kOnsetSpanMs * 0.001);
        s.refractory = static_cast<int>(sampleRate * kRefractoryMs * 0.001);
        s.searchLead = std::max(1, static_cast<int>(sampleRate * kSearchLeadMs * 0.001));
        s.cellLen = std::max(1, static_cast<int>(sampleRate * 0.001));
        s.coarseStep = std::max(1, roundToInt(sampleRate / kCoarseStepRate));
        s.hpA = std::exp(-2.0 * kPi * kDetectorHpfHz / sampleRate);
        s.smoothA = std::exp(-1.0 / (sampleRate * kDetectorSmoothMs * 0.001));

        const int maxCorr = static_cast<int>(sampleRate * kMaxCorrMs * 0.001);
        const int reach = windowSamples(Window::ms60, sampleRate) + maxCorr + s.fadeMaxLen + 2 * s.searchLead + 64;
        for (auto& r : s.rings)
            r.init(reach);
        s.control.init(reach);
        s.ref.assign(static_cast<size_t>(maxCorr), 0.0f);
        for (auto& r : s.dryRings)
            r.init(minDelaySamples(sampleRate) + 8);

        s.wetLowpass.prepare(sampleRate);
        s.wetLowpass.setType(LinkwitzRileyFilterType::lowpass);
        s.dryHighpass.prepare(sampleRate);
        s.dryHighpass.setType(LinkwitzRileyFilterType::highpass);
        s.dryAllpass.prepare(sampleRate);
        s.dryAllpass.setType(LinkwitzRileyFilterType::allpass);
        s.tonalityMix.reset(sampleRate, kBlendSeconds);
        s.setTonality(s.params.tonalityHz);

        s.wetMix.reset(sampleRate, kBlendSeconds);
        s.wetMix.setCurrentAndTargetValue(0.0f);
        s.mixSmoother.reset(sampleRate, kBlendSeconds);
        s.mixSmoother.setCurrentAndTargetValue(jlimit(0.0f, 1.0f, s.params.mix));
        s.running = s.enabled;
        s.applyWindow();
        s.reset();
    }

    void setEnabled(bool on)
    {
        auto& s = impl_;
        if (on == s.enabled)
            return;
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

    void setParams(const Params& p)
    {
        auto& s = impl_;
        if (p.window != s.params.window) {
            s.params.window = p.window;
            s.applyWindow();
            s.search = Impl::Search();
        }
        if (!exactlyEqual(p.semitones, s.params.semitones)) {
            s.params.semitones = p.semitones;
            const double ratio = std::pow(2.0, static_cast<double>(p.semitones) / 12.0);
            const auto direction = [](double r) { return r > 1.0 ? 1 : r < 1.0 ? -1 : 0; };
            if (direction(ratio) != direction(s.ratio) || std::abs(ratio - s.ratio) > kSearchKeepRatio)
                s.search = Impl::Search();
            s.ratio = ratio;
            s.updateFadeRange();
        }
        if (!exactlyEqual(p.tonalityHz, s.params.tonalityHz)) {
            s.params.tonalityHz = p.tonalityHz;
            s.setTonality(p.tonalityHz);
        }
        if (!exactlyEqual(p.mix, s.params.mix)) {
            s.params.mix = p.mix;
            s.mixSmoother.setTargetValue(jlimit(0.0f, 1.0f, p.mix));
        }
    }

    void process(float* const* channels, int numChannels, int numSamples)
    {
        auto& s = impl_;
        numChannels = std::min(numChannels, kMaxChannels);
        if (s.sampleRate <= 0.0 || numChannels <= 0 || numSamples <= 0 || !s.running)
            return;

        float* out[kMaxChannels];
        for (int ch = 0; ch < kMaxChannels; ++ch)
            out[ch] = channels[std::min(ch, numChannels - 1)];

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
            for (int ch = 0; ch < kMaxChannels; ++ch) {
                dry[ch] = out[ch][i];
                s.rings[static_cast<size_t>(ch)].write(now, dry[ch]);
                if (ch < numChannels)
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
                    if (s.search.ready && delay <= guard)
                        s.startFade(s.search.resultJump, s.fadeFor(s.search.bestNcc, s.fadeHi));
                }
            }
            if (s.search.active)
                s.stepSearch(s.search.perSample);

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
            if (s.primeLeft > 0 && --s.primeLeft == 0 && s.enabled)
                s.wetMix.setTargetValue(1.0f);

            const float powerFade = s.wetMix.getNextValue();
            const float mix = s.mixSmoother.getNextValue();
            const float tonalityMix = s.tonalityMix.getNextValue();
            for (int ch = 0; ch < numChannels; ++ch) {
                auto& ring = s.rings[static_cast<size_t>(ch)];
                float wet = ring.read(s.rA);
                if (s.fading)
                    wet = wet * gainA + gainB * ring.read(s.rB);
                const float dryRef = ring.at(now - s.dMin);
                auto& dryRing = s.dryRings[static_cast<size_t>(ch)];
                dryRing.write(now, s.dryHighpass.processSample(ch, dry[ch]));
                const float split = s.wetLowpass.processSample(ch, wet) + dryRing.at(now - s.dMin);
                wet += tonalityMix * (split - wet);
                const float dryMix = dryRef + tonalityMix * (s.dryAllpass.processSample(ch, dryRef) - dryRef);
                const float mixOut = dryMix + mix * (wet - dryMix);
                out[ch][i] = (1.0f - powerFade) * dry[ch] + powerFade * mixOut;
            }
            if (s.fading && s.fade >= 1.0) {
                s.rA = s.rB;
                s.fading = false;
            }
            s.rA += ratio;
            s.rB += ratio;
        }
        if (!s.enabled && !s.wetMix.isSmoothing())
            s.running = false;
    }

private:
    struct Impl
    {
        struct Ring
        {
            std::vector<float> buf;
            uint32_t mask = 0;

            void init(int minSize)
            {
                int size = 1;
                while (size < minSize)
                    size <<= 1;
                buf.assign(static_cast<size_t>(size), 0.0f);
                mask = static_cast<uint32_t>(size - 1);
            }

            void clear()
            {
                std::fill(buf.begin(), buf.end(), 0.0f);
            }

            void write(int64_t i, float x)
            {
                buf[static_cast<uint32_t>(i) & mask] = x;
            }

            float at(int64_t i) const
            {
                return buf[static_cast<uint32_t>(i) & mask];
            }

            float read(double pos) const
            {
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

        struct MirroredRing
        {
            std::vector<float> buf;
            uint32_t mask = 0;
            int size = 0;

            void init(int minSize)
            {
                size = 1;
                while (size < minSize)
                    size <<= 1;
                buf.assign(static_cast<size_t>(size) * 2, 0.0f);
                mask = static_cast<uint32_t>(size - 1);
            }

            void clear()
            {
                std::fill(buf.begin(), buf.end(), 0.0f);
            }

            void write(int64_t i, float x)
            {
                const auto k = static_cast<uint32_t>(i) & mask;
                buf[k] = x;
                buf[k + static_cast<uint32_t>(size)] = x;
            }

            const float* windowEndingAt(int64_t i, int n) const
            {
                const auto k = static_cast<uint32_t>(i) & mask;
                return buf.data() + (static_cast<int>(k) + size - n);
            }
        };

        double sampleRate = 0.0;
        Params params;
        double ratio = 1.0;

        bool enabled = false, running = false;
        LinearSmoothedValue wetMix;
        LinearSmoothedValue mixSmoother;
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

        struct Search
        {
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

        LinkwitzRileyFilter wetLowpass, dryHighpass, dryAllpass;
        LinearSmoothedValue tonalityMix;

        void applyWindow()
        {
            dMin = minDelaySamples(sampleRate);
            dMax = windowSamples(params.window, sampleRate);
            corrLen = std::min(dMax, static_cast<int>(sampleRate * kMaxCorrMs * 0.001));
            updateFadeRange();
        }

        void updateFadeRange()
        {
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

        int lowGuard() const
        {
            return std::max(dMin, static_cast<int>(std::max(0.0, ratio - 1.0) * fadeHi) + 4);
        }

        int fadeFor(double ncc, double room) const
        {
            const double t = jlimit(0.0, 1.0, (kFadeNccHi - ncc) / (kFadeNccHi - kFadeNccLo));
            const double len = fadeLo + t * (fadeHi - fadeLo);
            return std::max(8, static_cast<int>(std::min(len, room)));
        }

        void setTonality(float hz)
        {
            const bool on = hz > 0.0f;
            if (on) {
                const float cutoff = jlimit(kTonalityMinHz, kTonalityOffHz, hz);
                wetLowpass.setCutoffFrequency(cutoff);
                dryHighpass.setCutoffFrequency(cutoff);
                dryAllpass.setCutoffFrequency(cutoff);
            }
            tonalityMix.setTargetValue(on ? 1.0f : 0.0f);
        }

        void reset()
        {
            for (auto& r : rings)
                r.clear();
            control.clear();
            for (auto& r : dryRings)
                r.clear();
            rA = rB = static_cast<double>(written - dMin);
            primeLeft = dMin + 8;
            fading = false;
            search = Search();
            hpZ1 = hpZ2 = 0.0;
            eHf = 0.0;
            minHist.fill(1e9f);
            maxHist.fill(0.0f);
            minAcc = 1e9;
            maxAcc = 0.0;
            lastOnset = -(1 << 30);
        }

        double nccAt(int64_t b) const
        {
            const float* y = control.windowEndingAt(b, corrLen);
            const float* x = ref.data();
            float xy[8] = {}, yy[8] = {};
            int i = 0;
            for (; i + 8 <= corrLen; i += 8) {
                for (int k = 0; k < 8; ++k) {
                    xy[k] += x[i + k] * y[i + k];
                    yy[k] += y[i + k] * y[i + k];
                }
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

        void beginSearch(int lo, int hi, int lead)
        {
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
            for (int i = 0; i < corrLen; ++i)
                refXX += static_cast<double>(x[i]) * x[i];
            const int candidates = (search.hi - search.lo) / coarseStep + 1;
            search.perSample = lead > 0 ? (candidates + lead - 1) / lead : candidates;
        }

        void consider(int d)
        {
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

        void stepSearch(int count)
        {
            while (count-- > 0 && search.next <= search.hi) {
                consider(search.next);
                search.next += coarseStep;
            }
            if (search.next > search.hi) {
                if (coarseStep > 1) {
                    const int c = search.bestDelay;
                    for (int d = std::max(search.lo, c - coarseStep + 1);
                         d <= std::min(search.hi, c + coarseStep - 1);
                         ++d) {
                        if (d != c)
                            consider(d);
                    }
                }
                search.resultJump = (search.now0 - search.bestDelay) - search.a0;
                search.active = false;
                search.ready = true;
            }
        }

        void startFade(int64_t jump, int fadeSamples)
        {
            search.ready = false;
            const double destDelay = static_cast<double>(written - 1) - (rA + static_cast<double>(jump));
            if (destDelay < 2.0 || destDelay > static_cast<double>(rings[0].buf.size()) - 8.0)
                return;
            rB = rA + static_cast<double>(jump);
            fading = true;
            fade = 0.0;
            fadeInc = 1.0 / fadeSamples;
            fadeNcc = jlimit(0.0, 1.0, search.bestNcc);
        }

        bool detectOnset(double x)
        {
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

    Impl impl_;
};

class T3KPitchShiftPlugin : public Plugin
{
public:
    T3KPitchShiftPlugin()
        : Plugin(kParameterCount, 0, 0),
          fSemitones(0.0f),
          fStep(1.0f),
          fWindow(1.0f),
          fTonality(20000.0f),
          fMix(100.0f)
    {
        fShifter.setEnabled(true);
        fShifter.prepare(getSampleRate());
        applyCurrentParams();
    }

protected:
    const char* getLabel()       const override { return "T3K Pitch Shift"; }
    const char* getDescription() const override { return "Polyphonic correlation-spliced pitch shifter by TONE3000 (packaged for MOD by guky667)."; }
    const char* getMaker()       const override { return "TONE3000 (MOD build by guky667)"; }
    const char* getHomePage()    const override { return "https://github.com/tone-3000/tone3000-plugin"; }
    const char* getLicense()     const override { return "MIT"; }
    uint32_t    getVersion()     const override { return d_version(1, 0, 0); }
    int64_t     getUniqueId()    const override { return d_cconst('T', '3', 'p', 's'); }

    void initParameter(uint32_t index, Parameter& parameter) override
    {
        switch (index) {
        case kSemitones:
            parameter.hints      = kParameterIsAutomatable;
            parameter.name       = "Pitch";
            parameter.symbol     = "semitones";
            parameter.unit       = "st";
            parameter.ranges.def = 0.0f;
            parameter.ranges.min = -24.0f;
            parameter.ranges.max = 24.0f;
            break;
        case kStep:
            parameter.hints      = kParameterIsAutomatable | kParameterIsBoolean | kParameterIsInteger;
            parameter.name       = "Step";
            parameter.symbol     = "step";
            parameter.unit       = "";
            parameter.ranges.def = 1.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 1.0f;
            break;
        case kWindow:
            parameter.hints      = kParameterIsAutomatable | kParameterIsInteger;
            parameter.name       = "Window";
            parameter.symbol     = "window";
            parameter.unit       = "";
            parameter.ranges.def = 1.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 3.0f;
            break;
        case kTonality:
            parameter.hints      = kParameterIsAutomatable | kParameterIsLogarithmic;
            parameter.name       = "Tonality";
            parameter.symbol     = "tonality";
            parameter.unit       = "Hz";
            parameter.ranges.def = 20000.0f;
            parameter.ranges.min = 1000.0f;
            parameter.ranges.max = 20000.0f;
            break;
        case kMix:
            parameter.hints      = kParameterIsAutomatable;
            parameter.name       = "Mix";
            parameter.symbol     = "mix";
            parameter.unit       = "%";
            parameter.ranges.def = 100.0f;
            parameter.ranges.min = 0.0f;
            parameter.ranges.max = 100.0f;
            break;
        }
    }

    float getParameterValue(uint32_t index) const override
    {
        switch (index) {
        case kSemitones: return fSemitones;
        case kStep:      return fStep;
        case kWindow:    return fWindow;
        case kTonality:  return fTonality;
        case kMix:       return fMix;
        default:         return 0.0f;
        }
    }

    void setParameterValue(uint32_t index, float value) override
    {
        switch (index) {
        case kSemitones: fSemitones = jlimit(-24.0f, 24.0f, value); break;
        case kStep:      fStep      = value >= 0.5f ? 1.0f : 0.0f;  break;
        case kWindow:    fWindow    = jlimit(0.0f, 3.0f, value);    break;
        case kTonality:  fTonality  = jlimit(1000.0f, 20000.0f, value); break;
        case kMix:       fMix       = jlimit(0.0f, 100.0f, value);  break;
        }
    }

    void sampleRateChanged(double newSampleRate) override
    {
        fShifter.prepare(newSampleRate);
        applyCurrentParams();
    }

    void run(const float** inputs, float** outputs, uint32_t frames) override
    {
        applyCurrentParams();

        const float* in  = inputs[0];
        float*       out = outputs[0];
        if (out != in)
            std::copy(in, in + frames, out);

        float* channels[1] = { out };
        fShifter.process(channels, 1, static_cast<int>(frames));
    }

private:
    void applyCurrentParams()
    {
        PitchShift::Params p;
        p.semitones  = (fStep >= 0.5f) ? std::round(fSemitones) : fSemitones;
        p.window     = PitchShift::windowFromIndex(static_cast<int>(std::lround(fWindow)));
        p.tonalityHz = (fTonality >= 19990.0f) ? 0.0f : fTonality;
        p.mix        = fMix * 0.01f;
        fShifter.setParams(p);
    }

    float fSemitones;
    float fStep;
    float fWindow;
    float fTonality;
    float fMix;
    PitchShift fShifter;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(T3KPitchShiftPlugin)
};

Plugin* createPlugin() { return new T3KPitchShiftPlugin(); }

END_NAMESPACE_DISTRHO
endef

define T3K_PITCH_SHIFT_PLUGIN_INFO_H
#ifndef DISTRHO_PLUGIN_INFO_H_INCLUDED
#define DISTRHO_PLUGIN_INFO_H_INCLUDED

#define DISTRHO_PLUGIN_BRAND       "TONE3000"
#define DISTRHO_PLUGIN_NAME        "T3K Pitch Shift"
#define DISTRHO_PLUGIN_URI         "urn:mod-cookbook:t3k-pitch-shift"

#define DISTRHO_PLUGIN_HAS_UI       0
#define DISTRHO_PLUGIN_IS_RT_SAFE   1
#define DISTRHO_PLUGIN_NUM_INPUTS   1
#define DISTRHO_PLUGIN_NUM_OUTPUTS  1

enum Parameters {
    kSemitones = 0,
    kStep,
    kWindow,
    kTonality,
    kMix,
    kParameterCount
};

#endif
endef

define T3K_PITCH_SHIFT_PLUGIN_MAKEFILE
#!/usr/bin/make -f
NAME = t3k-pitch-shift
FILES_DSP = T3KPitchShiftPlugin.cpp
include ../../Makefile.plugins.mk
TARGETS = lv2_dsp
all: $$(TARGETS)
endef

define T3K_PITCH_SHIFT_MANIFEST_TTL
@prefix lv2:  <http://lv2plug.in/ns/lv2core#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<urn:mod-cookbook:t3k-pitch-shift>
    a lv2:Plugin , lv2:PitchPlugin ;
    lv2:binary <t3k-pitch-shift_dsp.so> ;
    rdfs:seeAlso <t3k-pitch-shift.ttl> , <modgui.ttl> .
endef

define T3K_PITCH_SHIFT_PLUGIN_TTL
@prefix doap:  <http://usefulinc.com/ns/doap#> .
@prefix foaf:  <http://xmlns.com/foaf/0.1/> .
@prefix lv2:   <http://lv2plug.in/ns/lv2core#> .
@prefix pprop: <http://lv2plug.in/ns/ext/port-props#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix units: <http://lv2plug.in/ns/extensions/units#> .

<urn:mod-cookbook:t3k-pitch-shift>
    a lv2:Plugin , lv2:PitchPlugin ;
    doap:name "T3K Pitch Shift" ;
    doap:license <http://opensource.org/licenses/MIT> ;
    doap:developer [
        foaf:name "TONE3000" ;
        foaf:homepage <https://github.com/tone-3000/tone3000-plugin>
    ] ;
    doap:maintainer [
        foaf:name "TONE3000 (MOD build by guky667)" ;
        foaf:homepage <https://github.com/tone-3000/tone3000-plugin>
    ] ;
    rdfs:comment "Polyphonic correlation-spliced pitch shifter with onset re-sync and Linkwitz-Riley tonality crossover by TONE3000 (https://github.com/tone-3000/tone3000-plugin), packaged for MOD by guky667." ;
    lv2:port [
        a lv2:InputPort , lv2:AudioPort ;
        lv2:index 0 ;
        lv2:symbol "in" ;
        lv2:name "Audio In"
    ] , [
        a lv2:OutputPort , lv2:AudioPort ;
        lv2:index 1 ;
        lv2:symbol "out" ;
        lv2:name "Audio Out"
    ] , [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 2 ;
        lv2:symbol "semitones" ;
        lv2:name "Pitch" ;
        lv2:default 0.0 ;
        lv2:minimum -24.0 ;
        lv2:maximum 24.0 ;
        units:unit units:semitone12TET
    ] , [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 3 ;
        lv2:symbol "step" ;
        lv2:name "Step" ;
        lv2:default 1 ;
        lv2:minimum 0 ;
        lv2:maximum 1 ;
        lv2:portProperty lv2:integer , lv2:toggled
    ] , [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 4 ;
        lv2:symbol "window" ;
        lv2:name "Window" ;
        lv2:default 1 ;
        lv2:minimum 0 ;
        lv2:maximum 3 ;
        lv2:portProperty lv2:integer , lv2:enumeration ;
        lv2:scalePoint [
            rdfs:label "20 ms (Guitar)" ;
            rdf:value 0
        ] , [
            rdfs:label "30 ms (Default / Bass)" ;
            rdf:value 1
        ] , [
            rdfs:label "40 ms" ;
            rdf:value 2
        ] , [
            rdfs:label "60 ms" ;
            rdf:value 3
        ]
    ] , [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 5 ;
        lv2:symbol "tonality" ;
        lv2:name "Tonality" ;
        lv2:default 20000.0 ;
        lv2:minimum 1000.0 ;
        lv2:maximum 20000.0 ;
        lv2:portProperty pprop:logarithmic ;
        units:unit units:hz
    ] , [
        a lv2:InputPort , lv2:ControlPort ;
        lv2:index 6 ;
        lv2:symbol "mix" ;
        lv2:name "Mix" ;
        lv2:default 100.0 ;
        lv2:minimum 0.0 ;
        lv2:maximum 100.0 ;
        units:unit units:pc
    ] .
endef

define T3K_PITCH_SHIFT_MODGUI_TTL
@prefix modgui: <http://moddevices.com/ns/modgui#> .
@prefix lv2:    <http://lv2plug.in/ns/lv2core#> .

<urn:mod-cookbook:t3k-pitch-shift>
    modgui:gui [
        modgui:resourcesDirectory <modgui> ;
        modgui:iconTemplate <modgui/icon.html> ;
        modgui:stylesheet <modgui/stylesheet.css> ;
        modgui:screenshot <modgui/screenshot.png> ;
        modgui:thumbnail <modgui/thumbnail.png> ;
        modgui:brand "TONE3000" ;
        modgui:label "T3K Pitch Shift" ;
        modgui:port [
            lv2:index 0 ;
            lv2:symbol "semitones" ;
            lv2:name "Pitch" ;
        ] , [
            lv2:index 1 ;
            lv2:symbol "step" ;
            lv2:name "Step" ;
        ] , [
            lv2:index 2 ;
            lv2:symbol "window" ;
            lv2:name "Window" ;
        ] , [
            lv2:index 3 ;
            lv2:symbol "tonality" ;
            lv2:name "Tonality" ;
        ] , [
            lv2:index 4 ;
            lv2:symbol "mix" ;
            lv2:name "Mix" ;
        ] ;
    ] .
endef

define T3K_PITCH_SHIFT_MODGUI_HTML
<div class="mod-pedal mod-pedal-{{{cns}}} t3k-pedal">
    <div mod-role="drag-handle" class="mod-drag-handle"></div>

    <!-- Unlabeled 1/4" TS Jacks -->
    <div class="mod-pedal-input">
        {{#effect.ports.audio.input}}
        <div class="mod-input mod-input-disconnected" title="{{name}}" mod-role="input-audio-port" mod-port-symbol="{{symbol}}">
            <div class="ts-jack left"></div>
        </div>
        {{/effect.ports.audio.input}}
    </div>

    <div class="mod-pedal-output">
        {{#effect.ports.audio.output}}
        <div class="mod-output mod-output-disconnected" title="{{name}}" mod-role="output-audio-port" mod-port-symbol="{{symbol}}">
            <div class="ts-jack right"></div>
        </div>
        {{/effect.ports.audio.output}}
    </div>

    <!-- Enclosure Body -->
    <div class="t3k-enclosure">
        <div class="screw tl"></div>
        <div class="screw tr"></div>
        <div class="screw bl"></div>
        <div class="screw br"></div>

        <!-- Top Credits Bar -->
        <div class="t3k-header">
            <div class="t3k-brand-wrap">
                <span class="t3k-dsp-badge">DSP: TONE3000</span>
                <span class="t3k-title">PITCH SHIFT</span>
            </div>
            <div class="t3k-packager">Duct-taped &amp; packaged by <strong>guky667</strong></div>
        </div>

        <!-- 5 Controls Row -->
        <div class="t3k-controls">
            <!-- 1. Pitch -->
            <div class="t3k-control-group">
                <div class="t3k-knob" mod-role="input-control-port" mod-port-symbol="semitones" mod-widget="film"></div>
                <div class="t3k-val" mod-role="input-control-value" mod-port-symbol="semitones"></div>
                <div class="t3k-label">Pitch</div>
            </div>

            <!-- 2. Step (Toggle Switch) -->
            <div class="t3k-control-group">
                <div class="t3k-switch" mod-role="input-control-port" mod-port-symbol="step" mod-widget="switch"></div>
                <div class="t3k-label">Step</div>
            </div>

            <!-- 3. Window (Rotary Selector) -->
            <div class="t3k-control-group">
                <div class="t3k-knob" mod-role="input-control-port" mod-port-symbol="window" mod-widget="film"></div>
                <div class="t3k-val" mod-role="input-control-value" mod-port-symbol="window"></div>
                <div class="t3k-label">Window</div>
            </div>

            <!-- 4. Tonality -->
            <div class="t3k-control-group">
                <div class="t3k-knob" mod-role="input-control-port" mod-port-symbol="tonality" mod-widget="film"></div>
                <div class="t3k-val" mod-role="input-control-value" mod-port-symbol="tonality"></div>
                <div class="t3k-label">Tonality</div>
            </div>

            <!-- 5. Mix -->
            <div class="t3k-control-group">
                <div class="t3k-knob" mod-role="input-control-port" mod-port-symbol="mix" mod-widget="film"></div>
                <div class="t3k-val" mod-role="input-control-value" mod-port-symbol="mix"></div>
                <div class="t3k-label">Mix</div>
            </div>
        </div>

        <!-- Bottom Bypass Bar -->
        <div class="t3k-footer">
            <div class="mod-light on" mod-role="bypass-light"></div>
            <div class="t3k-stomp" mod-role="bypass"></div>
        </div>
    </div>
</div>
endef

define T3K_PITCH_SHIFT_MODGUI_CSS
.mod-pedal-{{{cns}}}.t3k-pedal {
    position: relative;
    width: 620px;
    height: 290px;
    user-select: none;
    font-family: system-ui, -apple-system, sans-serif;
}

.mod-pedal-{{{cns}}} .mod-drag-handle {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1;
}

.mod-pedal-{{{cns}}} .t3k-enclosure {
    position: relative;
    width: 100%;
    height: 100%;
    border-radius: 18px;
    background:
        radial-gradient(circle at 16% 20%, rgba(139, 92, 246, 0.35), transparent 52%),
        radial-gradient(circle at 84% 78%, rgba(16, 185, 129, 0.28), transparent 52%),
        linear-gradient(135deg, #170d2a 0%, #111827 52%, #092218 100%);
    border: 2px solid #7c3aed;
    box-shadow:
        inset 0 0 24px rgba(16, 185, 129, 0.18),
        0 10px 26px rgba(0, 0, 0, 0.75);
    box-sizing: border-box;
    padding: 18px 26px 14px 26px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
}

/* Corner Screws */
.mod-pedal-{{{cns}}} .screw {
    position: absolute;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle, #9ca3af 20%, #374151 90%);
    border: 1px solid #111827;
    z-index: 2;
}
.mod-pedal-{{{cns}}} .screw.tl { top: 10px; left: 10px; }
.mod-pedal-{{{cns}}} .screw.tr { top: 10px; right: 10px; }
.mod-pedal-{{{cns}}} .screw.bl { bottom: 10px; left: 10px; }
.mod-pedal-{{{cns}}} .screw.br { bottom: 10px; right: 10px; }

/* Plain 1/4" TS Jacks */
.mod-pedal-{{{cns}}} .mod-pedal-input,
.mod-pedal-{{{cns}}} .mod-pedal-output {
    position: absolute;
    top: 118px;
    z-index: 5;
}
.mod-pedal-{{{cns}}} .mod-pedal-input { left: -16px; }
.mod-pedal-{{{cns}}} .mod-pedal-output { right: -16px; }

.mod-pedal-{{{cns}}} .ts-jack {
    width: 16px;
    height: 48px;
    background: linear-gradient(180deg, #6b7280 0%, #e5e7eb 50%, #4b5563 100%);
    border: 1px solid #1f2937;
}
.mod-pedal-{{{cns}}} .ts-jack.left { border-radius: 4px 0 0 4px; }
.mod-pedal-{{{cns}}} .ts-jack.right { border-radius: 0 4px 4px 0; }

/* Header & Credits */
.mod-pedal-{{{cns}}} .t3k-header {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(167, 139, 250, 0.28);
    padding-bottom: 8px;
}

.mod-pedal-{{{cns}}} .t3k-brand-wrap {
    display: flex;
    align-items: baseline;
    gap: 10px;
}

.mod-pedal-{{{cns}}} .t3k-dsp-badge {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1.4px;
    color: #34d399;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(52, 211, 153, 0.45);
    padding: 3px 8px;
    border-radius: 5px;
}

.mod-pedal-{{{cns}}} .t3k-title {
    font-size: 16px;
    font-weight: 900;
    letter-spacing: 2px;
    color: #f3f4f6;
}

.mod-pedal-{{{cns}}} .t3k-packager {
    font-size: 11px;
    font-family: monospace;
    color: #ddd6fe;
    background: rgba(139, 92, 246, 0.16);
    border: 1px dashed rgba(167, 139, 250, 0.5);
    padding: 3px 9px;
    border-radius: 4px;
    transform: rotate(-1deg);
}
.mod-pedal-{{{cns}}} .t3k-packager strong {
    color: #34d399;
}

/* Controls Row */
.mod-pedal-{{{cns}}} .t3k-controls {
    position: relative;
    z-index: 3;
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 8px 0;
}

.mod-pedal-{{{cns}}} .t3k-control-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 104px;
    gap: 5px;
}

.mod-pedal-{{{cns}}} .t3k-knob {
    width: 64px;
    height: 64px;
    background-image: url(/resources/knob-strip.svg{{{ns}}});
    background-repeat: no-repeat;
    background-size: 4160px 64px;
    background-position: 0 0;
    cursor: pointer;
}

.mod-pedal-{{{cns}}} .t3k-switch {
    width: 64px;
    height: 64px;
    background-image: url(/resources/switch-strip.svg{{{ns}}});
    background-repeat: no-repeat;
    background-size: 128px 64px;
    background-position: 0 0;
    cursor: pointer;
}

.mod-pedal-{{{cns}}} .t3k-val {
    min-height: 16px;
    font-size: 11px;
    font-family: monospace;
    color: #6ee7b7;
    background: rgba(6, 78, 59, 0.45);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 4px;
    padding: 1px 6px;
    text-align: center;
    white-space: nowrap;
}

.mod-pedal-{{{cns}}} .t3k-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #c4b5fd;
}

/* Footer Bypass Switch & LED */
.mod-pedal-{{{cns}}} .t3k-footer {
    position: relative;
    z-index: 3;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 14px;
    padding-top: 4px;
}

.mod-pedal-{{{cns}}} .mod-light {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 1px solid #1f2937;
}
.mod-pedal-{{{cns}}} .mod-light.on {
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
}
.mod-pedal-{{{cns}}} .mod-light.off {
    background: #3b0764;
    box-shadow: none;
}

.mod-pedal-{{{cns}}} .t3k-stomp {
    width: 44px;
    height: 24px;
    border-radius: 6px;
    background: linear-gradient(180deg, #e5e7eb 0%, #9ca3af 50%, #4b5563 100%);
    border: 2px solid #374151;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.6);
    cursor: pointer;
}
endef

export T3K_PITCH_SHIFT_PLUGIN_CPP
export T3K_PITCH_SHIFT_PLUGIN_INFO_H
export T3K_PITCH_SHIFT_PLUGIN_MAKEFILE
export T3K_PITCH_SHIFT_MANIFEST_TTL
export T3K_PITCH_SHIFT_PLUGIN_TTL
export T3K_PITCH_SHIFT_MODGUI_TTL
export T3K_PITCH_SHIFT_MODGUI_HTML
export T3K_PITCH_SHIFT_MODGUI_CSS

define T3K_PITCH_SHIFT_CONFIGURE_CMDS
	mkdir -p $(@D)/examples/t3k-pitch-shift
	printf '%s' "$$T3K_PITCH_SHIFT_PLUGIN_CPP"      > $(@D)/examples/t3k-pitch-shift/T3KPitchShiftPlugin.cpp
	printf '%s' "$$T3K_PITCH_SHIFT_PLUGIN_INFO_H"   > $(@D)/examples/t3k-pitch-shift/DistrhoPluginInfo.h
	printf '%s' "$$T3K_PITCH_SHIFT_PLUGIN_MAKEFILE" > $(@D)/examples/t3k-pitch-shift/Makefile
endef

define T3K_PITCH_SHIFT_BUILD_CMDS
	$(TARGET_MAKE_ENV) $(TARGET_CONFIGURE_OPTS) $(MAKE) NOOPT=true -C $(@D)/examples/t3k-pitch-shift lv2_dsp
endef

define T3K_PITCH_SHIFT_INSTALL_TARGET_CMDS
	mkdir -p $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui
	cp $(@D)/bin/t3k-pitch-shift.lv2/t3k-pitch-shift_dsp.so $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/
	printf '%s' "$$T3K_PITCH_SHIFT_MANIFEST_TTL" > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/manifest.ttl
	printf '%s' "$$T3K_PITCH_SHIFT_PLUGIN_TTL"   > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/t3k-pitch-shift.ttl
	printf '%s' "$$T3K_PITCH_SHIFT_MODGUI_TTL"   > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui.ttl
	printf '%s' "$$T3K_PITCH_SHIFT_MODGUI_HTML"  > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui/icon.html
	printf '%s' "$$T3K_PITCH_SHIFT_MODGUI_CSS"   > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui/stylesheet.css
	# Generate 65-frame rotary knob SVG strip (4160x64) for mod-widget="film"
	{ \
		printf '<svg xmlns="http://www.w3.org/2000/svg" width="4160" height="64" viewBox="0 0 4160 64">'; \
		i=0; \
		while [ $$i -lt 65 ]; do \
			x=$$((i * 64)); \
			cx=$$((x + 32)); \
			deg=$$((-135 + (i * 270) / 64)); \
			printf '<g transform="translate(%d,0)">' "$$x"; \
			printf '<circle cx="32" cy="32" r="27" fill="#181028" stroke="#7c3aed" stroke-width="2"/>'; \
			printf '<circle cx="32" cy="32" r="21" fill="#111827" stroke="#10b981" stroke-width="1.2" stroke-dasharray="96 36" transform="rotate(135 32 32)"/>'; \
			printf '</g>'; \
			printf '<g transform="rotate(%d %d 32)">' "$$deg" "$$cx"; \
			printf '<line x1="%d" y1="32" x2="%d" y2="11" stroke="#34d399" stroke-width="3" stroke-linecap="round"/>' "$$cx" "$$cx"; \
			printf '<circle cx="%d" cy="11" r="2.2" fill="#a7f3d0"/>' "$$cx"; \
			printf '</g>'; \
			i=$$((i + 1)); \
		done; \
		printf '</svg>\n'; \
	} > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui/knob-strip.svg
	# Generate 2-frame toggle switch SVG strip (128x64, Off/On without 0/1 text) for mod-widget="switch"
	{ \
		printf '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="64" viewBox="0 0 128 64">'; \
		printf '<g transform="translate(0,0)">'; \
		printf '<rect x="18" y="8" width="28" height="48" rx="14" fill="#181028" stroke="#7c3aed" stroke-width="2"/>'; \
		printf '<circle cx="32" cy="40" r="10" fill="#a78bfa"/>'; \
		printf '<text x="32" y="62" font-family="sans-serif" font-size="9" font-weight="bold" fill="#a78bfa" text-anchor="middle">OFF</text>'; \
		printf '</g>'; \
		printf '<g transform="translate(64,0)">'; \
		printf '<rect x="18" y="8" width="28" height="48" rx="14" fill="#092218" stroke="#10b981" stroke-width="2"/>'; \
		printf '<circle cx="32" cy="22" r="10" fill="#34d399"/>'; \
		printf '<text x="32" y="62" font-family="sans-serif" font-size="9" font-weight="bold" fill="#34d399" text-anchor="middle">ON</text>'; \
		printf '</g>'; \
		printf '</svg>\n'; \
	} > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui/switch-strip.svg
	# Create minimal 1x1 PNG placeholders for screenshot.png and thumbnail.png
	printf 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==' | base64 -d > $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui/screenshot.png
	cp $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui/screenshot.png $($(PKG)_PKGDIR)/t3k-pitch-shift.lv2/modgui/thumbnail.png
endef

$(eval $(generic-package))