#include "app.hpp"

#include "calendar.hpp"
#include "earth_texture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

namespace wildlife {
namespace {

constexpr std::array speed_factors{0.01, 0.1,  0.25, 1.0,   2.0,
                                   4.0,  16.0, 64.0, 256.0, 1024.0};
constexpr const char *speed_labels[] = {
    "0.01x", "0.1x", "0.25x", "1x", "2x", "4x", "16x", "64x", "256x", "1024x"};
constexpr std::array trail_hours{0.0,
                                 1.0,
                                 3.0,
                                 6.0,
                                 12.0,
                                 24.0,
                                 48.0,
                                 96.0,
                                 7.0 * 24,
                                 14.0 * 24,
                                 30.0 * 24,
                                 90.0 * 24,
                                 182.625 * 24,
                                 365.25 * 24,
                                 2 * 365.25 * 24,
                                 5 * 365.25 * 24,
                                 10 * 365.25 * 24,
                                 25 * 365.25 * 24,
                                 50 * 365.25 * 24};
constexpr const char *trail_labels[] = {
    "Off",     "1 hour",   "3 hours",  "6 hours", "12 hours",
    "1 day",   "2 days",   "4 days",   "1 week",  "2 weeks",
    "1 month", "3 months", "6 months", "1 year",  "2 years",
    "5 years", "10 years", "25 years", "50 years"};
static_assert(trail_hours.size() == std::size(trail_labels));
constexpr double simulated_hours_per_second = 6.0;
constexpr float wheel_zoom_factor = 1.24f;
constexpr float click_zoom_factor = 1.65f;
constexpr double overview_latitude = 24.0;
constexpr double overview_longitude = -24.0;
constexpr double earth_radius_km = 6371.0;
constexpr double radians_to_degrees = 180.0 / std::numbers::pi;
constexpr double rotation_degrees_per_second = 0.1 * radians_to_degrees;
constexpr ImU32 ink = IM_COL32(232, 241, 247, 255);
constexpr ImU32 muted = IM_COL32(153, 174, 189, 255);
constexpr ImU32 panel = IM_COL32(8, 15, 23, 225);

float distance_squared(ImVec2 a, ImVec2 b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return dx * dx + dy * dy;
}

MapLabelRect map_label(ImDrawList &draw, ImVec2 origin, const char *title,
                       const char *detail) {
  const ImVec2 a = ImGui::CalcTextSize(title);
  const ImVec2 b = ImGui::CalcTextSize(detail);
  const ImVec2 max{origin.x + std::max(a.x, b.x) + 20.0f,
                   origin.y + a.y + b.y + 17.0f};
  draw.AddRectFilled(origin, max, panel, 4.0f);
  draw.AddText({origin.x + 10.0f, origin.y + 7.0f}, ink, title);
  draw.AddText({origin.x + 10.0f, origin.y + a.y + 10.0f}, muted, detail);
  return {origin, max};
}

MapLabelRect draw_scale(ImDrawList &draw, ImVec2 origin, double scale) {
  const double distance = 110.0 / scale;
  const double magnitude = std::pow(10.0, std::floor(std::log10(distance)));
  const double digit = distance / magnitude;
  const double km = magnitude * (digit >= 5.0 ? 5.0 : digit >= 2.0 ? 2.0 : 1.0);
  const auto width = static_cast<float>(km * scale);
  char text[32];
  if (km < .000001)
    std::snprintf(text, sizeof(text), "%g um", km * 1.0e9);
  else if (km < .001)
    std::snprintf(text, sizeof(text), "%g mm", km * 1.0e6);
  else if (km < 1.0)
    std::snprintf(text, sizeof(text), "%g m", km * 1000.0);
  else
    std::snprintf(text, sizeof(text), "%g km", km);
  const ImVec2 size = ImGui::CalcTextSize(text);
  const MapLabelRect bounds{
      {origin.x - 8.0f, origin.y - size.y - 12.0f},
      {origin.x + std::max(width, size.x) + 8.0f, origin.y + 9.0f}};
  draw.AddRectFilled(bounds.min, bounds.max, panel, 4.0f);
  draw.AddText({origin.x, origin.y - size.y - 6.0f}, ink, text);
  draw.AddLine(origin, {origin.x + width, origin.y}, ink, 1.5f);
  draw.AddLine({origin.x, origin.y - 3.0f}, {origin.x, origin.y + 3.0f}, ink);
  draw.AddLine({origin.x + width, origin.y - 3.0f},
               {origin.x + width, origin.y + 3.0f}, ink);
  return bounds;
}

} // namespace

App::App(const std::filesystem::path &executable)
    : enabled_(simulation_.species().size(), 1) {
  listed_species_.reserve(enabled_.size());
  if (ImGui::GetCurrentContext()) {
    habitat_font_ = load_habitat_font(executable);
    grain_sprite_ = make_grain_sprite(*ImGui::GetIO().Fonts);
  }
}

void App::draw() {
  const auto &io = ImGui::GetIO();
  forecast_.poll();
  forecast_.tick(simulation_);
  if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
    paused_ = !paused_;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    following_ = false;
    selected_.reset();
  }
  if (!paused_ && forecast_.predicted()) {
    const double frame_seconds =
        std::min(static_cast<double>(io.DeltaTime), 0.1);
    const double advance = frame_seconds * simulated_hours_per_second *
                           speed_factors[speed_index_];
    if (forecast_.ready())
      forecast_.seek_relative(simulation_, advance);
  }

  const auto *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::PushFont(habitat_font_, 15.0f);
  constexpr auto flags = ImGuiWindowFlags_NoDecoration |
                         ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoSavedSettings;
  ImGui::Begin("Stray", nullptr, flags);
  draw_toolbar();
  ImGui::Separator();
  draw_sidebar();
  ImGui::SameLine();
  forecast_.select(simulation_, selected_);
  draw_map();
  ImGui::End();
  ImGui::PopFont();
}

void App::draw_toolbar() {
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("STRAY");
  ImGui::SameLine();
  ImGui::TextDisabled("|  Animal migration forecasts");
  ImGui::BeginDisabled(!forecast_.predicted());
  ImGui::SameLine();
  if (ImGui::Button(paused_ ? "Play" : "Pause"))
    paused_ = !paused_;
  ImGui::SetItemTooltip("Space");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(76.0f);
  ImGui::Combo("##speed", &speed_index_, speed_labels,
               IM_ARRAYSIZE(speed_labels));
  ImGui::SetItemTooltip(
      "At 1x, one second advances six simulated hours at every zoom level.");
  ImGui::SameLine();
  ImGui::BeginDisabled(!paused_);
  if (ImGui::Button("+1 hour")) {
    if (forecast_.ready())
      forecast_.seek_relative(simulation_, 1.0);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button(forecast_.scenario() ? "Restart month" : "Restart")) {
    if (forecast_.ready())
      forecast_.seek_relative(simulation_, -1.0e10);
    paused_ = true;
  }
  ImGui::EndDisabled();
}

void App::draw_sidebar() {
  const float width =
      std::clamp(ImGui::GetContentRegionAvail().x * 0.29f, 222.0f, 320.0f);
  ImGui::BeginChild("##sidebar", {width, 0.0f}, ImGuiChildFlags_Borders);

  if (const auto animal = forecast_.controls(simulation_, enabled_)) {
    selected_ = animal;
    paused_ = !forecast_.scenario();
    following_ = false;
  }
  if (selected_ && selected_->colony >= simulation_.colonies().size()) {
    selected_.reset();
    following_ = false;
  }
  const auto forecast_clock = forecast_.clock(simulation_.elapsed_hours());
  if (forecast_clock) {
    ImGui::SeparatorText("Forecast clock (UTC)");
    const auto date =
        calendar_detail::read_date(forecast_clock->hours, forecast_clock->year);
    ImGui::Text("%02d %s %d  %02d:%02d", date.day,
                calendar_detail::months[date.month], forecast_clock->year,
                date.minute_of_day / 60, date.minute_of_day % 60);
  }
  ImGui::SetNextItemWidth(105.0f);
  ImGui::Combo("Trail history", &trail_index_, trail_labels,
               IM_ARRAYSIZE(trail_labels));
  ImGui::SetItemTooltip("Fading grains along each animal's timed route. Long "
                        "gaps in observed GPS are left blank.");

  ImGui::SeparatorText("Observed species / taxa");
  const auto &species = simulation_.species();
  const auto &colonies = simulation_.colonies();
  ImGui::TextDisabled("%zu species  /  %zu groups", species.size(),
                      colonies.size());
  if (ImGui::SmallButton("All"))
    std::fill(enabled_.begin(), enabled_.end(), 1);
  ImGui::SameLine();
  if (ImGui::SmallButton("None"))
    std::fill(enabled_.begin(), enabled_.end(), 0);
  ImGui::SameLine();
  if (ImGui::SmallButton("Color key"))
    ImGui::OpenPopup("##color_key");
  if (ImGui::BeginPopup("##color_key")) {
    ImGui::TextUnformatted("Species have distinct shades; individuals vary subtly.");
    ImGui::TextUnformatted("Related genera share a main color.");
    ImGui::Separator();
    for (const auto &family : animal_color_key()) {
      ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(family.color), "%s",
                         family.label);
    }
    ImGui::EndPopup();
  }
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::InputTextWithHint("##species_filter", "Search species",
                               species_filter_.InputBuf,
                               sizeof(species_filter_.InputBuf))) {
    species_filter_.Build();
  }
  listed_species_.clear();
  for (std::size_t i = 0; i < species.size(); ++i) {
    const auto &item = species[i];
    if (!species_filter_.IsActive()) {
      listed_species_.push_back(i);
      continue;
    }
    char searchable[160];
    std::snprintf(searchable, sizeof(searchable), "%s %.*s", item.name.data(),
                  static_cast<int>(item.scientific_name.size()),
                  item.scientific_name.empty() ? ""
                                               : item.scientific_name.data());
    if (species_filter_.PassFilter(searchable))
      listed_species_.push_back(i);
  }
  const float list_height =
      std::clamp(ImGui::GetContentRegionAvail().y * 0.53f, 65.0f, 340.0f);
  ImGui::BeginChild("##species", {0.0f, list_height}, ImGuiChildFlags_Borders);
  ImGuiListClipper rows;
  rows.Begin(static_cast<int>(listed_species_.size()));
  while (rows.Step()) {
    for (int row = rows.DisplayStart; row < rows.DisplayEnd; ++row) {
      const auto i = listed_species_[row];
      ImGui::PushID(static_cast<int>(i));
      bool enabled = enabled_[i] != 0;
      if (ImGui::Checkbox("##enabled", &enabled))
        enabled_[i] = enabled;
      ImGui::SameLine();
      const bool active =
          std::any_of(colonies.begin(), colonies.end(),
                      [&](const Colony &c) { return c.species == i; });
      ImGui::BeginDisabled(!active);
      if (ImGui::SmallButton("Find"))
        locate_species(i);
      ImGui::EndDisabled();
      if (!active)
        ImGui::SetItemTooltip("No recent GPS history at this issue time. "
                              "Choose another forecast date.");
      ImGui::SameLine();
      const auto group = species[i].group;
      ImGui::TextColored(
          ImGui::ColorConvertU32ToFloat4(species_color(species[i])), "%s",
          species[i].name.data());
      if (species[i].scientific_name.empty()) {
        ImGui::SetItemTooltip(
            "%s\n%s", species[i].name.data(),
            animal_color_key()[static_cast<std::size_t>(group)].label);
      } else {
        ImGui::SetItemTooltip("%s\n%s", species[i].name.data(),
                              species[i].scientific_name.data());
      }
      ImGui::PopID();
    }
  }
  ImGui::EndChild();

  ImGui::SeparatorText("Inspect");
  if (selected_) {
    const auto &colony = colonies[selected_->colony];
    const auto current =
        simulation_.sample_animal(*selected_, simulation_.elapsed_hours());
    ImGui::TextWrapped("%s", species[colony.species].name.data());
    ImGui::TextDisabled("%s", colony.region.data());
    ImGui::TextWrapped("%s",
                       simulation_.is_forecast(selected_->colony)
                           ? "Model forecast | interpolated between horizons"
                           : "Observed GPS history");
    ImGui::Text("Tracked animals in cohort: %d", colony.population);
    const auto id = std::string(simulation_.animal_name(*selected_));
    ImGui::Text("Tag: %s", id.substr(id.find_last_of(':') + 1).c_str());
    ImGui::SetItemTooltip("%s", id.c_str());

    if (ImGui::SmallButton("Zoom to group")) {
      following_ = false;
      focus_group(selected_->colony);
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Follow", &following_) && following_)
      focus(current.position, std::max(10.0f, globe_zoom_));
  } else {
    ImGui::TextWrapped("Every marker is an observed animal. Load GPS histories "
                       "or choose a forecast date to explore the cohorts.");
  }
  ImGui::SeparatorText("Earth");
  if (ImGui::SmallButton("Alps detail")) {
    following_ = false;
    terrain_tilt_ = true;
    focus(earth_position(45.9764f, 7.6586f), 2450.0f);
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Map sources"))
    ImGui::OpenPopup("##map_sources");
  if (ImGui::BeginPopup("##map_sources")) {
    ImGui::TextLinkOpenURL("NASA Earth Observatory",
                           "https://science.nasa.gov/earth/earth-observatory/"
                           "earth-at-night/maps/");
    ImGui::TextLinkOpenURL("EOxCloudless by EOX IT Services GmbH",
                           "https://cloudless.eox.at");
    ImGui::TextDisabled("Contains modified Copernicus Sentinel data 2016");
    ImGui::TextLinkOpenURL(
        "Mapzen elevation data and credits",
        "https://github.com/tilezen/joerd/blob/master/docs/attribution.md");
    ImGui::TextLinkOpenURL(
        "NOAA global elevation and seafloor relief",
        "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/"
        "iso?id=gov.noaa.ngdc.mgg.dem%3A301");
    ImGui::EndPopup();
  }
  ImGui::Checkbox("Angled terrain", &terrain_tilt_);
  ImGui::SetItemTooltip("Keep a gently tilted map view when zoomed in. "
                        "Right-drag to adjust the angle.");
  ImGui::EndChild();
}

void App::draw_map() {
  ImGui::BeginChild("##map_panel", {0.0f, 0.0f}, ImGuiChildFlags_Borders,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);
  if (ImGui::Button("Reset view"))
    reset_camera();
  ImGui::SameLine();
  ImGui::Checkbox("Auto-rotate", &auto_rotate_);
  ImGui::SetItemTooltip(
      "One turn in about 63 seconds at overview scale. Slows as you zoom in.");
  ImGui::SameLine();
  if (ImGui::Checkbox("Both hemispheres", &both_hemispheres_))
    following_ = false;
  ImGui::SameLine();
  ImGui::Checkbox("Night view", &night_view_);
  ImGui::SetItemTooltip(
      "Turn the map toward the night side to inspect city lights.");
  if (ImGui::GetContentRegionAvail().x > 610.0f) {
    ImGui::SameLine();
    ImGui::TextDisabled("Drag: pan | Right-drag: tilt | Scroll: zoom");
  }

  const ImVec2 size = ImGui::GetContentRegionAvail();
  if (size.x > 2.0f && size.y > 2.0f) {
    const ImVec2 min = ImGui::GetCursorScreenPos();
    const ImVec2 max{min.x + size.x, min.y + size.y};
    const float radius =
        std::clamp(std::min(size.x, size.y) * 0.15f, 56.0f, 103.0f);
    const ImVec2 center{max.x - radius - 16.0f, max.y - radius - 16.0f};
    const auto &io = ImGui::GetIO();
    ImGui::InvisibleButton("##map_canvas", size);
    const bool hovered = ImGui::IsItemHovered();
    const bool over_calendar =
        hovered &&
        distance_squared(io.MousePos, center) <= radius * radius * 1.05f;
    const double hours = simulation_.elapsed_hours();
    const auto clock = forecast_.clock(hours);
    const double calendar_hours = clock ? clock->hours : hours;
    const int calendar_year = clock ? clock->year : 0;
    const auto calendar_hit =
        over_calendar && clock
            ? calendar_scrub_hours(center, radius, io.MousePos, calendar_hours,
                                   false, calendar_year)
            : std::nullopt;
    if (ImGui::IsItemActivated()) {
      map_dragged_ = false;
      zoomed_on_click_ = false;
      calendar_dragging_ = calendar_hit.has_value();
      drag_opposite_ =
          both_hemispheres_ && io.MousePos.x > (min.x + max.x) * 0.5f;
    }
    if (calendar_dragging_ &&
        (ImGui::IsItemActive() || ImGui::IsItemDeactivated())) {
      if (const auto hour =
              calendar_scrub_hours(center, radius, io.MousePos, calendar_hours,
                                   true, calendar_year)) {
        if (clock)
          forecast_.seek_relative(simulation_, *hour - clock->hours);
        else
          simulation_.seek(*hour);
        paused_ = true;
      }
    } else if (ImGui::IsItemActive() &&
               ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      map_dragged_ = true;
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    }
    auto &draw = *ImGui::GetWindowDrawList();
    draw.PushClipRect(min, max, true);
    const float calendar_clearance = radius + 6.0f;
    const MapLabelRect calendar{
        {center.x - calendar_clearance, center.y - calendar_clearance},
        {center.x + calendar_clearance, center.y + calendar_clearance}};
    draw_globe(draw, min, max, calendar,
               hovered && !over_calendar && !calendar_dragging_);
    const auto updated_clock = forecast_.clock(simulation_.elapsed_hours());
    if (updated_clock)
      draw_calendar(draw, center, radius, updated_clock->hours,
                    CalendarRecording{updated_clock->year, updated_clock->begin,
                                      updated_clock->end});

    draw.PopClipRect();
    if (calendar_hit && !calendar_dragging_) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
      ImGui::SetTooltip("Drag the ring to move through the simulated year.");
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      map_dragged_ = false;
      calendar_dragging_ = false;
    }
  }
  ImGui::EndChild();
}

void App::draw_globe(ImDrawList &draw, ImVec2 min, ImVec2 max,
                     MapLabelRect calendar, bool hovered) {
  const auto &io = ImGui::GetIO();
  const double now = ImGui::GetTime();
  if (map_dragged_ || calendar_dragging_ || (hovered && io.MouseWheel != 0.0f))
    rotation_resume_time_ = now + 4.0;
  if (following_ && selected_) {
    const auto current =
        simulation_.sample_animal(*selected_, simulation_.elapsed_hours());
    view_latitude_ = std::atan2(double(current.position.y),
                                std::hypot(double(current.position.x),
                                           double(current.position.z))) *
                     radians_to_degrees;
    view_longitude_ =
        std::atan2(double(current.position.z), double(current.position.x)) *
        radians_to_degrees;
  } else if (auto_rotate_ && now >= rotation_resume_time_) {
    const double zoom_weight =
        both_hemispheres_ ? 1.0
                          : std::clamp((2.5 - globe_zoom_) / 1.5, 0.0, 1.0);
    const double angle = rotation_degrees_per_second * zoom_weight *
                         std::min(io.DeltaTime, 0.1f);
    view_longitude_ = std::remainder(view_longitude_ + angle, 360.0);
  }
  const ImVec2 center{(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f};
  const float base_radius = std::min(max.x - min.x, max.y - min.y) * 0.425f;
  const float long_side = std::max(max.x - min.x, max.y - min.y);
  const auto tilt_for = [&](float zoom) {
    return terrain_tilt_
               ? terrain_angle_ *
                     std::clamp((base_radius * zoom / long_side - 2.0f) * .5f,
                                0.0f, 1.0f)
               : 0.0f;
  };
  const float quarter_width = (max.x - min.x) * 0.25f;
  const float pair_radius = std::min(quarter_width, base_radius) * 0.87f;
  const ImVec2 left{center.x - quarter_width, center.y};
  const ImVec2 right{center.x + quarter_width, center.y};

  if (hovered && io.MouseWheel != 0.0f) {
    const float factor = std::pow(wheel_zoom_factor, io.MouseWheel);
    if (both_hemispheres_ && io.MouseWheel > 0.0f) {
      const bool opposite = io.MousePos.x > center.x;
      const GlobeView view(opposite ? right : left, pair_radius,
                           opposite ? -view_latitude_ : view_latitude_,
                           view_longitude_ + (opposite ? 180.0 : 0.0));
      if (view.unproject(io.MousePos, view_latitude_, view_longitude_)) {
        globe_zoom_ = std::min(factor, max_zoom);
        target_zoom_ = globe_zoom_;
        both_hemispheres_ = false;
        following_ = false;
      }
    } else if (!both_hemispheres_) {
      target_zoom_ = std::clamp(target_zoom_ * factor, min_zoom, max_zoom);
    }
  }
  if (!both_hemispheres_ && target_zoom_ != globe_zoom_) {
    const float zoom =
        std::lerp(globe_zoom_, target_zoom_,
                  1.0f - std::exp(-15.0f * std::min(io.DeltaTime, .1f)));
    if (zoom != globe_zoom_) {
      const GlobeView before(center, base_radius * globe_zoom_, view_latitude_,
                             view_longitude_, tilt_for(globe_zoom_));
      const GlobeView after(center, base_radius * zoom, view_latitude_,
                            view_longitude_, tilt_for(zoom));
      double latitude = 0.0;
      double longitude = 0.0;
      const ImVec2 anchor = hovered ? io.MousePos : center;
      if (before.unproject(anchor, latitude, longitude)) {
        after.center_for_anchor(anchor, latitude, longitude, view_latitude_,
                                view_longitude_);
      }
      globe_zoom_ = zoom;
      if (std::abs(target_zoom_ - globe_zoom_) < globe_zoom_ * .0001f)
        globe_zoom_ = target_zoom_;
      following_ = false;
    }
  }
  if (hovered && !both_hemispheres_ && globe_zoom_ > 6 &&
      ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
    terrain_angle_ =
        std::clamp(terrain_angle_ + io.MouseDelta.y * .16f, 0.f, 35.f);
  }
  const bool dual_view = both_hemispheres_;
  const float zoom = dual_view ? 1.0f : globe_zoom_;
  const float radius = dual_view ? pair_radius : base_radius * zoom;
  const float tilt = dual_view ? 0.0f : tilt_for(zoom);
  if (ImGui::IsItemActive() && map_dragged_) {
    following_ = false;
    const double sensitivity = radians_to_degrees / std::max(radius, 1.0f);
    const double longitude_scale =
        std::max(0.22, std::cos(view_latitude_ / radians_to_degrees));
    view_longitude_ = std::remainder(
        view_longitude_ - io.MouseDelta.x * sensitivity / longitude_scale,
        360.0);
    const double vertical_delta =
        dual_view && drag_opposite_ ? -io.MouseDelta.y : io.MouseDelta.y;
    const double pitch_scale = std::cos(tilt / radians_to_degrees);
    view_latitude_ =
        std::clamp(view_latitude_ + vertical_delta * sensitivity / pitch_scale,
                   -90.0, 90.0);
  }

  draw.AddRectFilledMultiColor(
      min, max, IM_COL32(12, 16, 22, 255), IM_COL32(7, 11, 16, 255),
      IM_COL32(5, 8, 12, 255), IM_COL32(8, 12, 18, 255));
  const GlobeView globe(dual_view ? left : center, radius, view_latitude_,
                        view_longitude_, tilt);
  const GlobeView opposite(right, radius, -view_latitude_,
                           view_longitude_ + 180.0);
  const auto sun = GlobeView::sunlight;
  const Vec3 light = night_view_ ? Vec3{-sun.x, -sun.y, -sun.z} : sun;
  globe.draw(draw, light);
  if (dual_view) {
    // The opposite camera reverses its east and forward axes.
    opposite.draw(draw, {-light.x, light.y, -light.z});
  }
  const bool can_pick = hovered && !map_dragged_;
  const auto pointer = can_pick ? std::optional{io.MousePos} : std::nullopt;
  const bool pick_opposite = dual_view && io.MousePos.x > center.x;
  const float pixels_per_km = static_cast<float>(radius / earth_radius_km);
  const double history = trail_hours[trail_index_];
  auto nearest = draw_populations(
      draw, globe, simulation_, enabled_, pixels_per_km, history, selected_,
      pick_opposite ? std::nullopt : pointer, grain_sprite_);
  if (dual_view) {
    const auto other = draw_populations(
        draw, opposite, simulation_, enabled_, pixels_per_km, history,
        selected_, pick_opposite ? pointer : std::nullopt, grain_sprite_);
    if (other)
      nearest = other;
  }
  forecast_.draw(draw, globe, simulation_);
  if (dual_view)
    forecast_.draw(draw, opposite, simulation_);
  if (can_pick && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    const float closer = zoom * click_zoom_factor;
    if (nearest) {
      selected_ = nearest;
      following_ = false;
      const auto current =
          simulation_.sample_animal(*nearest, simulation_.elapsed_hours());
      focus(current.position, std::max(8.0f, closer));
      zoomed_on_click_ = true;
    } else {
      double latitude = 0.0;
      double longitude = 0.0;
      const auto &clicked = pick_opposite ? opposite : globe;
      if (clicked.unproject(io.MousePos, latitude, longitude)) {
        selected_.reset();
        following_ = false;
        focus(earth_position(static_cast<float>(latitude),
                             static_cast<float>(longitude)),
              closer);
        zoomed_on_click_ = true;
      }
    }
  } else if (nearest) {
    const auto &colony = simulation_.colonies()[nearest->colony];
    const auto &species = simulation_.species()[colony.species];
    if (!zoomed_on_click_ && ImGui::IsItemDeactivated() &&
        ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      following_ = false;
      selected_ = nearest;
    } else if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
      ImGui::SetTooltip("%s\n%s | Animal %zu", species.name.data(),
                        colony.region.data(), nearest->individual + 1);
    }
  }
  std::array<MapLabelRect, 4> reserved{calendar};
  std::size_t reserved_count = 1;
  char level[96];
  std::snprintf(level, sizeof(level), "%s  |  Scroll to explore",
                forecast_.predicted() ? "Model-predicted animals"
                : forecast_.ready()   ? "Observed animals"
                                      : "Loading observed wildlife...");
  reserved[reserved_count++] = map_label(draw, {min.x + 12.0f, min.y + 12.0f},
                                         "WILDLIFE MOVEMENT", level);
  if (earth_detail_visible()) {
    reserved[reserved_count++] =
        map_label(draw, {min.x + 12.0f, min.y + 60.0f},
                  "EOxCloudless by EOX IT Services GmbH",
                  "Modified Copernicus Sentinel 2016  |  Mapzen elevation");
  }
  if (!dual_view) {
    reserved[reserved_count++] =
        draw_scale(draw, {min.x + 21.0f, max.y - 20.0f}, pixels_per_km);
    draw_habitat_labels(draw, globe, habitat_font_, {min, max}, zoom,
                        std::span{reserved}.first(reserved_count));
  }
}

void App::reset_camera() {
  both_hemispheres_ = false;
  following_ = false;
  view_latitude_ = overview_latitude;
  view_longitude_ = overview_longitude;
  globe_zoom_ = 1.0f;
  target_zoom_ = globe_zoom_;
  rotation_resume_time_ = ImGui::GetTime() + 4.0;
}

void App::focus(Vec3 position, float zoom) {
  both_hemispheres_ = false;
  view_latitude_ =
      std::atan2(double(position.y),
                 std::hypot(double(position.x), double(position.z))) *
      radians_to_degrees;
  view_longitude_ =
      std::atan2(double(position.z), double(position.x)) * radians_to_degrees;
  globe_zoom_ = std::clamp(zoom, min_zoom, max_zoom);
  target_zoom_ = globe_zoom_;
  rotation_resume_time_ = ImGui::GetTime() + 4.0;
}

void App::test_trail_history(double hours) {
  const auto found =
      std::lower_bound(trail_hours.begin(), trail_hours.end(), hours);
  trail_index_ = int(std::min<std::size_t>(found - trail_hours.begin(),
                                           trail_hours.size() - 1));
}
void App::test_zoom(float zoom) {
  globe_zoom_ = std::clamp(zoom, min_zoom, max_zoom);
  target_zoom_ = globe_zoom_;
}

void App::focus_group(std::size_t colony) {
  std::vector<AnimalSample> samples(simulation_.colonies()[colony].population);
  const auto count =
      simulation_.sample_colony(colony, simulation_.elapsed_hours(), samples);
  if (count == 0)
    return;
  double x = 0, y = 0, z = 0;
  for (std::size_t i = 0; i < count; ++i) {
    x += samples[i].position.x;
    y += samples[i].position.y;
    z += samples[i].position.z;
  }
  const double length = std::hypot(x, y, z);
  if (length < 1e-9) {
    focus(simulation_.colonies()[colony].position, 1.0f);
    return;
  }
  x /= length;
  y /= length;
  z /= length;
  double extent_km = 2.0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto p = samples[i].position;
    const double chord = std::hypot(p.x - x, p.y - y, p.z - z);
    extent_km = std::max(extent_km, 2 * earth_radius_km *
                                        std::asin(std::min(1., chord * .5)));
  }
  focus({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)},
        static_cast<float>(.52 * earth_radius_km / extent_km));
}

void App::locate_species(std::size_t species) {
  const auto &colonies = simulation_.colonies();
  auto group = colonies.end();
  for (auto candidate = colonies.begin(); candidate != colonies.end();
       ++candidate) {
    if (candidate->species == species &&
        (group == colonies.end() ||
         candidate->population > group->population)) {
      group = candidate;
    }
  }
  if (group == colonies.end())
    return;
  enabled_[species] = 1;
  selected_ = AnimalId{static_cast<std::size_t>(group - colonies.begin()), 0};
  following_ = false;
  focus_group(selected_->colony);
}

bool App::focus_dots(std::string_view name, float zoom) {
  const auto &catalog = simulation_.species();
  if (name.empty()) {
    if (simulation_.colonies().empty())
      return false;
    focus_population();
    test_zoom(zoom);
    auto_rotate_ = false;
    return true;
  }
  const auto species =
      std::find_if(catalog.begin(), catalog.end(), [&](const Species &item) {
        return name.empty() || item.name == name ||
               item.scientific_name == name;
      });
  if (species == catalog.end())
    return false;
  locate_species(static_cast<std::size_t>(species - catalog.begin()));
  if (!selected_ || simulation_.colonies()[selected_->colony].species !=
                        std::size_t(species - catalog.begin()))
    return false;
  focus(simulation_.sample_animal(*selected_, simulation_.elapsed_hours())
            .position,
        zoom);
  auto_rotate_ = false;
  return true;
}

bool App::load_forecast(const std::string &path) {
  if (!forecast_.load(path))
    return false;
  try {
    selected_ = forecast_.apply(simulation_, enabled_);
    return true;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Invalid forecast: %s\n", error.what());
    return false;
  }
}

void App::focus_population() {
  const auto &colonies = simulation_.colonies();
  if (colonies.empty())
    return;
  const auto group = std::max_element(colonies.begin(), colonies.end(),
                                      [](const Colony &a, const Colony &b) {
                                        return a.population < b.population;
                                      });
  selected_ = AnimalId{static_cast<std::size_t>(group - colonies.begin()), 0};
  focus_group(selected_->colony);
}

} // namespace wildlife
