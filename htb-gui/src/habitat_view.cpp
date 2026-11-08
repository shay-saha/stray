#include "habitat_view.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>

namespace wildlife {
namespace {

constexpr float label_size = 18.0f;

struct HabitatSite {
  const char *name;
  float latitude;
  float longitude;
  float min_zoom;
};

// Representative label points from Natural Earth and the atlas's geographic
// sites.
constexpr HabitatSite sites[] = {
    {"Lake Chad", 13.07f, 14.49f, 1.24f},
    {"Sinai", 29.50f, 33.85f, 1.32f},
    {"Rift Valley", -0.71f, 36.20f, 0.78f},
    {"Amazon Basin", -7.02f, -61.94f, 0.70f},
    {"Sahara", 21.53f, 11.89f, 0.70f},
    {"Himalayas", 29.46f, 85.05f, 0.70f},
    {"Congo Basin", -1.07f, 21.45f, 0.79f},
    {"Rocky Mountains", 46.17f, -113.15f, 0.70f},
    {"Alps", 46.44f, 10.24f, 0.76f},
    {"Great Barrier Reef", -18.20f, 147.50f, 0.82f},
    {"Gobi Desert", 43.03f, 106.25f, 0.82f},
    {"Patagonia", -43.42f, -69.39f, 0.75f},
    {"Sargasso Sea", 33.9f, -62.3f, 0.79f},
    {"Borneo", 0.86f, 114.21f, 0.86f},
    {"Canadian Boreal Forest", 57.8f, -106.7f, 0.80f},
    {"Antarctic Peninsula", -67.6f, -65.0f, 0.79f},
    {"Kalahari Desert", -23.61f, 20.31f, 1.04f},
    {"Gulf of Alaska", 52.2f, -143.5f, 0.84f},
    {"Tibetan Plateau", 33.38f, 90.30f, 1.10f},
    {"Great Plains", 42.9f, -103.09f, 0.82f},
    {"Western Ghats", 15.54f, 74.69f, 1.30f},
    {"Tasman Sea", -39.2f, 159.5f, 0.87f},
    {"Philippine Sea", 18.4f, 131.7f, 0.90f},
    {"Arctic Archipelago", 73.6f, -90.1f, 0.83f},
    {"Ross Sea", -74.2f, 175.0f, 0.89f},
    {"Weddell Sea", -73.8f, -38.6f, 0.92f},
    {"Central Australia", -24.4f, 133.3f, 0.78f},
    {"Bering Strait", 66.1f, -168.9f, 1.10f},
    {"Lake Baikal", 53.41f, 107.70f, 1.25f},
    {"Pantanal", -18.02f, -57.11f, 1.30f},
    {"Okavango Delta", -19.3f, 22.8f, 1.28f},
    {"Carpathians", 47.45f, 22.98f, 1.32f},
    {"Atlas Mountains", 31.3f, -6.4f, 1.32f},
    {"Sundarbans", 22.4f, 89.17f, 1.35f},
    {"Greater Caucasus", 43.01f, 43.75f, 1.36f},
    {"Appalachians", 39.9f, -79.4f, 1.35f},
    {"Valdivian Forest", -39.6f, -72.2f, 1.43f},
    {"Scottish Highlands", 56.9f, -4.7f, 1.46f},
    {"Pyrenees", 42.74f, 0.64f, 1.50f},
    {"Serengeti", -2.31f, 34.40f, 1.60f},
    {"Lake Victoria", -1.10f, 32.90f, 1.72f},
    {"Namib Desert", -21.33f, 14.16f, 1.53f},
    {"Madagascar Highlands", -18.6f, 47.8f, 1.44f},
    {"Lake Tanganyika", -6.21f, 29.88f, 1.66f},
    {"Benguela Current", -29.7f, 7.7f, 1.50f},
    {"California Current", 34.5f, -128.4f, 1.45f},
    {"Grand Banks", 45.0f, -53.0f, 1.53f},
    {"Svalbard", 78.65f, 21.77f, 1.37f},
    {"Altai Mountains", 47.72f, 90.51f, 1.48f},
    {"Mekong Delta", 10.29f, 105.97f, 1.63f},
    {"Lena Delta", 72.98f, 126.32f, 1.68f},
    {"Mongolian Steppe", 46.5f, 105.2f, 1.77f},
    {"Japanese Alps", 36.5f, 137.7f, 1.70f},
    {"New Guinea Highlands", -6.1f, 145.0f, 1.64f},
    {"Southern Alps", -43.96f, 169.66f, 1.58f},
    {"Tasmania", -42.1f, 146.3f, 1.55f},
    {"Orinoco Basin", 4.8f, -70.2f, 1.47f},
    {"Cerrado", -12.4f, -47.7f, 1.54f},
    {"Atlantic Forest", -24.2f, -49.4f, 1.61f},
    {"Galapagos Islands", -0.67f, -90.55f, 2.00f},
    {"Colorado Plateau", 36.8f, -112.0f, 1.75f},
    {"Greater Yellowstone", 44.6f, -110.5f, 1.81f},
    {"Florida Everglades", 25.55f, -80.86f, 1.82f},
    {"Yucatan Peninsula", 20.3f, -89.3f, 1.63f},
    {"Chihuahuan Desert", 29.7f, -105.3f, 1.78f},
    {"Coral Sea", -18.0f, 155.5f, 1.90f},
    {"Lake Turkana", 3.58f, 36.18f, 2.45f},
    {"Zambezi Delta", -18.48f, 36.14f, 2.35f},
    {"Bialowieza Forest", 52.73f, 23.81f, 2.60f},
    {"Matterhorn", 45.9764f, 7.6586f, 55.0f},
    {"Yellowstone Lake", 44.446f, -110.346f, 23.0f},
    {"Victoria Falls", -17.928f, 25.853f, 25.0f}};

const auto &locations() {
  static const auto positions = [] {
    std::array<Vec3, std::size(sites)> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = earth_position(sites[i].latitude, sites[i].longitude);
    }
    return result;
  }();
  return positions;
}

float fade(float first, float last, float value) {
  const float fraction =
      std::clamp((value - first) / (last - first), 0.0f, 1.0f);
  return fraction * fraction * (3.0f - 2.0f * fraction);
}

float separation(MapLabelRect a, MapLabelRect b) {
  return std::max({b.min.x - a.max.x, a.min.x - b.max.x, b.min.y - a.max.y,
                   a.min.y - b.max.y});
}

void draw_anchored_text(ImDrawList &draw, ImFont *font, ImVec2 origin,
                        ImU32 color, const char *name) {
  const int first = draw.VtxBuffer.Size;
  draw.AddText(font, label_size, origin, color, name);
  // Restore the fractional position snapped by ImGui so the lettering moves
  // with the marker.
  const float dx = origin.x - std::trunc(origin.x);
  const float dy = origin.y - std::trunc(origin.y);
  for (int i = first; i < draw.VtxBuffer.Size; ++i) {
    draw.VtxBuffer[i].pos.x += dx;
    draw.VtxBuffer[i].pos.y += dy;
  }
}

} // namespace

ImFont *load_habitat_font(const std::filesystem::path &executable) {
  constexpr auto filename = "Urbanist-Variable.ttf";
  const auto bin = executable.parent_path();
  const std::array paths{bin / filename,
                         bin.parent_path() / "Resources" / filename,
                         std::filesystem::path("assets") / filename};
  for (const auto &path : paths) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error))
      continue;
    auto &atlas = *ImGui::GetIO().Fonts;
    if (atlas.Fonts.empty())
      atlas.AddFontDefault();
    return atlas.AddFontFromFileTTF(path.string().c_str(), label_size);
  }
  return nullptr;
}

void draw_habitat_labels(ImDrawList &draw, const GlobeView &globe, ImFont *font,
                         MapLabelRect bounds, float zoom,
                         std::span<const MapLabelRect> blocked) {
  const float closeup = fade(1.55f, 1.82f, zoom);
  if (closeup == 0.0f)
    return;
  ImGui::PushFont(
      font,
      label_size); // Bake the font at the active viewport's pixel density.
  font = ImGui::GetFont();
  struct VisibleLabel {
    MapLabelRect area;
    float alpha;
  };
  std::array<VisibleLabel, std::size(sites)> occupied;
  std::size_t placed = 0;
  const auto &positions = locations();
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const auto &site = sites[i];
    const float reveal =
        std::min(closeup, fade(site.min_zoom, site.min_zoom * 1.22f, zoom));
    if (reveal == 0.0f)
      continue;
    ImVec2 anchor;
    float facing = 0.0f;
    if (!globe.project(positions[i], anchor, &facing) ||
        anchor.x < bounds.min.x || anchor.x > bounds.max.x ||
        anchor.y < bounds.min.y || anchor.y > bounds.max.y)
      continue;
    float alpha = reveal * fade(0.10f, 0.37f, facing);
    if (alpha == 0.0f)
      continue;

    const auto text = font->CalcTextSizeA(label_size, 1000.0f, 0.0f, site.name);
    const ImVec2 origin{anchor.x + 5.5f, anchor.y - text.y * 0.5f};
    const MapLabelRect area{
        {anchor.x - 3.0f, origin.y - 3.0f},
        {origin.x + text.x + 3.0f, origin.y + text.y + 3.0f}};
    const std::array corners{area.min, area.max, ImVec2{area.min.x, area.max.y},
                             ImVec2{area.max.x, area.min.y}};
    float edge =
        std::min({area.min.x - bounds.min.x, bounds.max.x - area.max.x,
                  area.min.y - bounds.min.y, bounds.max.y - area.max.y});
    for (const auto corner : corners)
      edge = std::min(edge, globe.edge_clearance(corner));
    alpha *= fade(0.0f, 20.0f, edge);

    for (const auto overlay : blocked)
      alpha *= fade(0.0f, 10.0f, separation(area, overlay));
    if (alpha == 0.0f)
      continue;
    // Earlier sites have priority. Soften nearby labels without changing their
    // positions.
    for (const auto &other : std::span{occupied}.first(placed)) {
      alpha *= std::lerp(1.0f, fade(0.0f, 9.0f, separation(area, other.area)),
                         other.alpha);
    }
    if (alpha == 0.0f)
      continue;
    occupied[placed++] = {area, alpha};

    const auto shadow = IM_COL32(7, 11, 13, static_cast<int>(215.0f * alpha));
    const auto ink = IM_COL32(232, 232, 225, static_cast<int>(238.0f * alpha));
    draw.AddCircleFilled(anchor, 2.35f, shadow, 12);
    draw.AddCircleFilled(anchor, 1.4f, ink, 12);
    draw_anchored_text(draw, font, {origin.x + 0.75f, origin.y + 0.75f}, shadow,
                       site.name);
    draw_anchored_text(draw, font, origin, ink, site.name);
  }
  ImGui::PopFont();
}

} // namespace wildlife
