#pragma once

#include <imgui.h>
#include <span>

namespace wildlife {

struct GlowTrailPoint {
  ImVec2 position;
  float opacity;
};

// One batch per map/frame. The callback runs after CPU geometry is populated,
// before the animal markers, using the same clip and display coordinates.
bool initialize_glow_trails();
void begin_glow_trails(ImDrawList &draw);
void add_glow_trail(std::span<const GlowTrailPoint> points, ImU32 color,
                    float half_width);
void shutdown_glow_trails();

} // namespace wildlife
