#pragma once
#include <algorithm>
#include <array>
#include <memory>

class PitchShift {
public:
  static constexpr int kMaxChannels = 2;
  static constexpr int kSemitoneRange = 24;  // knob is ±24
  static constexpr float kTonalityMinHz = 1000.0f;
  static constexpr float kTonalityOffHz = 20000.0f;

  enum class Window { ms20, ms30, ms40, ms60 };
  static constexpr std::array<int, 4> kWindowMs{20, 30, 40, 60};
  static constexpr Window kDefaultWindow = Window::ms30;
  static constexpr double kMinDelayMs = 2.0;

  static int windowMs(Window w) { return kWindowMs[static_cast<size_t>(w)]; }
  static Window windowFromIndex(int index) {
    return static_cast<Window>(std::clamp(index, 0, static_cast<int>(kWindowMs.size()) - 1));
  }

  struct Params {
    int semitones = 0;
    // Frequency above which the dry bypasses the shifter. 0 or >= 20000: off.
    float tonalityHz = 0.0f;
    Window window = kDefaultWindow;
  };

  PitchShift();
  ~PitchShift();

  void prepare(double sampleRate, int maxBlockSamples);
  void setEnabled(bool on);
  bool isRunning() const;
  void setParams(const Params& p);

  void process(float* const* channels, int numChannels, int numSamples);
  void processMono(float* inout, int numSamples) { process(&inout, 1, numSamples); }

  int latencySamples() const;
  static int latencySamples(Window w, double sampleRate) {
    return (minDelaySamples(sampleRate) + windowSamples(w, sampleRate)) / 2;
  }
  static double latencyMs(Window w) { return (kMinDelayMs + windowMs(w)) * 0.5; }

  static int minDelaySamples(double sampleRate) { return static_cast<int>(sampleRate * kMinDelayMs * 0.001); }
  static int windowSamples(Window w, double sampleRate) {
    return static_cast<int>(sampleRate * windowMs(w) * 0.001);
  }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};