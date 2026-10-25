#include "PitchShift.h"
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__has_include) && __has_include(<lv2/core/lv2.h>)
  #include <lv2/core/lv2.h>
#else
  extern "C" {
    typedef void* LV2_Handle;
    typedef struct { const char* URI; void* data; } LV2_Feature;
    typedef struct _LV2_Descriptor {
      const char* URI;
      LV2_Handle (*instantiate)(const struct _LV2_Descriptor*, double, const char*, const LV2_Feature* const*);
      void (*connect_port)(LV2_Handle, uint32_t, void*);
      void (*activate)(LV2_Handle);
      void (*run)(LV2_Handle, uint32_t);
      void (*deactivate)(LV2_Handle);
      void (*cleanup)(LV2_Handle);
      const void* (*extension_data)(const char*);
    } LV2_Descriptor;
    #if defined(_WIN32)
      #define LV2_SYMBOL_EXPORT __declspec(dllexport)
    #else
      #define LV2_SYMBOL_EXPORT __attribute__((visibility("default")))
    #endif
  }
#endif

enum PortIndex : uint32_t {
  PORT_IN = 0,
  PORT_OUT = 1,
  PORT_SEMITONES = 2,
  PORT_TONALITY_HZ = 3
};

struct PitchShiftLV2 {
  PitchShift dsp;
  double sampleRate = 48000.0;
  const float* in = nullptr;
  float* out = nullptr;
  const float* semitones = nullptr;
  const float* tonalityHz = nullptr;
};

static LV2_Handle instantiate(const LV2_Descriptor*, double rate, const char*, const LV2_Feature* const*) {
    auto* self = new PitchShiftLV2();
    self->sampleRate = rate;
    self->dsp.prepare(rate, 4096);
    self->dsp.setEnabled(false); // Start bypassed until a non-zero pitch is set
    return static_cast<LV2_Handle>(self);
}

static void connect_port(LV2_Handle instance, uint32_t port, void* data) {
  auto* self = static_cast<PitchShiftLV2*>(instance);
  switch (port) {
    case PORT_IN:          self->in = static_cast<const float*>(data); break;
    case PORT_OUT:         self->out = static_cast<float*>(data); break;
    case PORT_SEMITONES:   self->semitones = static_cast<const float*>(data); break;
    case PORT_TONALITY_HZ: self->tonalityHz = static_cast<const float*>(data); break;
  }
}

static void activate(LV2_Handle instance) {
    auto* self = static_cast<PitchShiftLV2*>(instance);
    self->dsp.prepare(self->sampleRate, 4096);
    self->dsp.setEnabled(false);
}

static void run(LV2_Handle instance, uint32_t n_samples) {
    auto* self = static_cast<PitchShiftLV2*>(instance);
    if (!self->in || !self->out || n_samples == 0) return;

    PitchShift::Params p;
    p.semitones = self->semitones ? static_cast<int>(std::lround(*self->semitones)) : 0;
    p.tonalityHz = self->tonalityHz ? *self->tonalityHz : PitchShift::kTonalityOffHz;
    p.window = PitchShift::Window::ms30;

    self->dsp.setParams(p);
    // Automatically crossfade to 0-latency dry bypass when Pitch is 0
    self->dsp.setEnabled(p.semitones != 0);

    if (self->in != self->out) {
        std::memcpy(self->out, self->in, n_samples * sizeof(float));
    }
    self->dsp.processMono(self->out, static_cast<int>(n_samples));
}

static void cleanup(LV2_Handle instance) {
  delete static_cast<PitchShiftLV2*>(instance);
}

static const LV2_Descriptor descriptor = {
  "http://lv2plug.in/plugins/pitch_shift",
  instantiate,
  connect_port,
  activate,
  run,
  nullptr,
  cleanup,
  nullptr
};

extern "C" LV2_SYMBOL_EXPORT const LV2_Descriptor* lv2_descriptor(uint32_t index) {
  return (index == 0) ? &descriptor : nullptr;
}