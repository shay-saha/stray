#include "population_view.hpp"
#include "trail_view.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace wildlife {
namespace {

constexpr float earth_radius_km = 6371.0f;
constexpr std::array palette{
    AnimalColorFamily{"Birds", IM_COL32(210, 173, 122, 255)},
    AnimalColorFamily{"Grazers and browsers", IM_COL32(156, 177, 139, 255)},
    AnimalColorFamily{"Land carnivorans", IM_COL32(192, 133, 115, 255)},
    AnimalColorFamily{"Primates", IM_COL32(191, 141, 170, 255)},
    AnimalColorFamily{"Other land mammals", IM_COL32(183, 160, 148, 255)},
    AnimalColorFamily{"Aquatic mammals", IM_COL32(154, 165, 211, 255)},
    AnimalColorFamily{"Fish and krill", IM_COL32(118, 181, 170, 255)},
    AnimalColorFamily{"Reptiles", IM_COL32(163, 149, 194, 255)}};
static_assert(palette.size() == static_cast<std::size_t>(AnimalGroup::Count));
constexpr ImU32 ivory = IM_COL32(228, 222, 206, 255);

ImU32 blend(ImU32 from, ImU32 to, float weight) {
  const auto channel = [&](int shift) {
    const float a = static_cast<float>((from >> shift) & 255u);
    const float b = static_cast<float>((to >> shift) & 255u);
    return static_cast<unsigned int>(std::lerp(a, b, weight));
  };
  return IM_COL32(channel(IM_COL32_R_SHIFT), channel(IM_COL32_G_SHIFT),
                  channel(IM_COL32_B_SHIFT), 255);
}

ImU32 opacity(ImU32 color, float alpha) {
  return (color & ~IM_COL32_A_MASK) |
         (static_cast<ImU32>(alpha) << IM_COL32_A_SHIFT);
}

bool inside(ImVec2 point, ImVec2 min, ImVec2 max, float margin) {
  return point.x >= min.x - margin && point.x <= max.x + margin &&
         point.y >= min.y - margin && point.y <= max.y + margin;
}

float distance_squared(ImVec2 a, ImVec2 b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return dx * dx + dy * dy;
}

} // namespace

ImFontAtlasRectId make_grain_sprite(ImFontAtlas &atlas) {
  constexpr int size = 32;
  ImFontAtlasRect region;
  const auto id = atlas.AddCustomRect(size, size, &region);
  if (id == ImFontAtlasRectId_Invalid)
    return id;
  auto &texture = *atlas.TexData;
  for (int y = 0; y < size; ++y) {
    const float v = 2.0f * (y + 0.5f) / size - 1.0f;
    for (int x = 0; x < size; ++x) {
      const float u = 2.0f * (x + 0.5f) / size - 1.0f;
      const float falloff = std::max(0.0f, 1.0f - u * u - v * v);
      auto *pixel = static_cast<unsigned char *>(
          texture.GetPixelsAt(region.x + x, region.y + y));
      std::fill_n(pixel, texture.BytesPerPixel - 1, 255);
      pixel[texture.BytesPerPixel - 1] =
          static_cast<unsigned char>(std::lround(255.0f * falloff * falloff));
    }
  }
  return id;
}

std::span<const AnimalColorFamily> animal_color_key() { return palette; }

ImU32 species_color(AnimalGroup group, std::size_t species) {
  const ImU32 base = palette[static_cast<std::size_t>(group)].color;
  constexpr std::array shades{-0.15f, -0.08f, 0.0f, 0.10f, 0.20f};
  const float shade = shades[species % shades.size()];
  return blend(base, shade < 0.0f ? IM_COL32(65, 66, 73, 255) : ivory,
               std::abs(shade));
}

std::optional<AnimalId> draw_populations(
    ImDrawList &draw, const GlobeView &view, const Simulation &simulation,
    std::span<const std::uint8_t> enabled, float pixels_per_km,
    double history_hours, std::optional<AnimalId> selected,
    std::optional<ImVec2> pointer, ImFontAtlasRectId grain_sprite) {
  const ImVec2 min = draw.GetClipRectMin();
  const ImVec2 max = draw.GetClipRectMax();
  const float detail = std::clamp((pixels_per_km - 0.05f) / 0.24f, 0.0f, 1.0f);
  const double now = simulation.elapsed_hours();
  const auto &colonies = simulation.colonies();
  const auto &species = simulation.species();
  const auto &atlas = *ImGui::GetIO().Fonts;
  ImFontAtlasRect grain;
  const bool has_sprite = atlas.GetCustomRect(grain_sprite, &grain);
  std::vector<AnimalSample> heads;
  std::vector<ImVec2> positions;
  std::vector<float> facing;
  std::vector<AnimalTrailSegment> history;
  std::vector<AnimalTrailPoint> long_history;
  std::vector<GlowTrailPoint> stroke;
  if (history_hours > 0)
    begin_glow_trails(draw);
  std::optional<AnimalId> nearest;
  float nearest_distance = 8.0f * 8.0f;

  for (std::size_t i = 0; i < colonies.size(); ++i) {
    const auto &colony = colonies[i];
    if (!enabled[colony.species])
      continue;
    ImVec2 center;
    float depth = 0.0f;
    const bool center_visible = view.project(colony.position, center, &depth);
    const float bound = colony.draw_radius_km;
    if (depth < -bound / earth_radius_km ||
        (center_visible &&
         !inside(center, min, max, bound * pixels_per_km + 4.0f)))
      continue;

    const auto size = static_cast<std::size_t>(colony.population);
    heads.resize(size);
    positions.resize(size);
    facing.resize(size);
    const auto count = simulation.sample_colony(i, now, heads);
    bool show_history = false;
    for (std::size_t j = 0; j < count; ++j) {
      if (!view.project(heads[j].position, positions[j], &facing[j]) ||
          !inside(positions[j], min, max, 1.5f)) {
        facing[j] = -1.0f;
        continue;
      }
      show_history |=
          heads[j].activity == Activity::Traveling || pixels_per_km > 0.20f;
    }

    if (show_history && history_hours > 0.0) {
      for (std::size_t j = 0; j < count; ++j) {
        if (facing[j] < 0.0f || (heads[j].activity != Activity::Traveling &&
                                 pixels_per_km <= 0.20f))
          continue;
        const AnimalId animal{i, j};
        const ImU32 color = species_color(species[colony.species].group, colony.species);
        stroke.clear();
        const auto flush = [&] {
          add_glow_trail(stroke, color, 3.8f + 1.6f * detail);
          stroke.clear();
        };
        const auto point_at = [&](Vec3 p, double hour) {
          ImVec2 at;
          float facing = 0;
          if (!view.project(p, at, &facing)) {
            flush();
            return;
          }
          const float fade =
              float(std::clamp(1 - (now - hour) / history_hours, 0., 1.));
          const float alpha = (.70f + .12f * detail) * fade * fade *
                              std::clamp(facing * 4.f, 0.f, 1.f);
          if (!stroke.empty() &&
              distance_squared(stroke.back().position, at) < .36f)
            return;
          stroke.push_back({at, alpha});
        };
        const auto segment = [&](Vec3 a, Vec3 b, double ta, double tb) {
          ImVec2 pa, pb;
          const bool va = view.project(a, pa), vb = view.project(b, pb);
          if (!va && !vb) {
            flush();
            return;
          }
          if (va && vb &&
              ((pa.x < min.x - 32 && pb.x < min.x - 32) ||
               (pa.x > max.x + 32 && pb.x > max.x + 32) ||
               (pa.y < min.y - 32 && pb.y < min.y - 32) ||
               (pa.y > max.y + 32 && pb.y > max.y + 32))) {
            flush();
            return;
          }
          const double chord = std::hypot(double(a.x - b.x), double(a.y - b.y),
                                          double(a.z - b.z));
          const double distance =
              2 * earth_radius_km * std::asin(std::min(1., chord * .5));
          // Subdivide on the globe, so a long route follows Earth's curvature
          // rather than drawing a straight screen-space shortcut at the limb.
          const int steps = std::clamp(
              int(std::ceil(distance * pixels_per_km / 16.)), 1, 64);
          for (int step = 1; step <= steps; ++step) {
            const double t = double(step) / steps;
            const double hour = std::lerp(ta, tb, t);
            Vec3 p{float(std::lerp(double(a.x), double(b.x), t)),
                   float(std::lerp(double(a.y), double(b.y), t)),
                   float(std::lerp(double(a.z), double(b.z), t))};
            const double length =
                std::hypot(double(p.x), double(p.y), double(p.z));
            if (length < 1e-6) {
              flush();
              continue;
            }
            p = {float(p.x / length), float(p.y / length), float(p.z / length)};
            point_at(p, hour);
          }
        };
        if (history_hours > 96) {
          const bool focused = selected && *selected == animal;
          simulation.sample_trail_points(
              animal, now, history_hours,
              focused ? 128 : std::size_t(8 + 24 * detail), long_history);
          Vec3 previous = heads[j].position;
          double previous_hour = now;
          point_at(previous, previous_hour);
          for (const auto &point : long_history) {
            if (point.connected_to_newer)
              segment(previous, point.position, previous_hour, point.hour);
            else {
              flush();
              point_at(point.position, point.hour);
            }
            previous = point.position;
            previous_hour = point.hour;
          }
        } else {
          simulation.sample_trail(animal, now, history_hours, history);
          double previous_hour = -1e100;
          for (const auto &part : history) {
            if (std::abs(previous_hour - part.start_hour) > 1e-6) {
              flush();
              point_at(part.start, part.start_hour);
            }
            segment(part.start, part.end, part.start_hour, part.end_hour);
            previous_hour = part.end_hour;
          }
        }
        flush();
      }
    }

    for (std::size_t j = 0; j < count; ++j) {
      if (facing[j] < 0.0f)
        continue;
      const AnimalId animal{i, j};
      const auto point = positions[j];
      const auto &head = heads[j];
      const ImU32 color =
          species_color(species[colony.species].group, colony.species);
      const ImU32 highlight = blend(color, ivory, .18f);
      const float limb = std::clamp(facing[j] * 4.0f, 0.0f, 1.0f);
      const float radius =
          std::lerp(1.10f, 1.95f, head.brightness) + .65f * detail;
      const float alpha = std::lerp(90.0f, 245.0f, head.brightness) * limb;
      const ImU32 tint = opacity(
          head.activity == Activity::Traveling ? highlight : color, alpha);
      if (has_sprite) {
        const float halo = radius * 3.0f;
        draw.AddImage(atlas.TexRef, {point.x - halo, point.y - halo},
                      {point.x + halo, point.y + halo}, grain.uv0, grain.uv1,
                      opacity(color, (62 + 42 * detail) * limb));
        draw.AddImage(atlas.TexRef, {point.x - radius, point.y - radius},
                      {point.x + radius, point.y + radius}, grain.uv0,
                      grain.uv1, tint);
      } else {
        draw.AddCircleFilled(point, radius * 0.48f, tint, 8);
      }
      if (pointer) {
        const float distance = distance_squared(point, *pointer);
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest = animal;
        }
      }
    }
  }

  if (selected && enabled[colonies[selected->colony].species]) {
    ImVec2 position;
    if (view.project(simulation.sample_animal(*selected, now).position,
                     position) &&
        inside(position, min, max, 5.0f)) {
      const auto index = colonies[selected->colony].species;
      draw.AddCircle(
          position, 5.0f,
          species_color(species[index].group, index), 16,
          1.2f);
    }
  }
  return nearest;
}

} // namespace wildlife
